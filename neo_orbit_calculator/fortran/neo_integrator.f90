module neo_integrator
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: qp = selected_real_kind(33, 4931)
  integer, parameter :: nbody = 27
  integer(c_int), parameter :: body_ids(nbody) = [ &
       10_c_int, 1_c_int, 2_c_int, 399_c_int, 301_c_int, 4_c_int, &
       5_c_int, 6_c_int, 7_c_int, 8_c_int, 9_c_int, &
       2000001_c_int, 2000004_c_int, 2000002_c_int, 2000010_c_int, &
       2000511_c_int, 2000704_c_int, 2000052_c_int, 2000087_c_int, &
       2000015_c_int, 2000003_c_int, 2000016_c_int, 2000107_c_int, &
       2000088_c_int, 2000007_c_int, 2000031_c_int, 2000065_c_int ]
  real(qp), parameter :: au_km = 149597870.700_qp
  real(qp), parameter :: day_s = 86400.0_qp
  real(qp), parameter :: c_km_s = 299792.458_qp
  real(qp), parameter :: solar_pressure = 4.56e-6_qp

  interface
    subroutine spkez_c(target, et, ref, abcorr, observer, state, light_time) &
        bind(C, name="spkez_c")
      import :: c_int, c_double, c_char
      integer(c_int), value :: target
      real(c_double), value :: et
      character(kind=c_char), intent(in) :: ref(*)
      character(kind=c_char), intent(in) :: abcorr(*)
      integer(c_int), value :: observer
      real(c_double), intent(out) :: state(6)
      real(c_double), intent(out) :: light_time
    end subroutine spkez_c
  end interface

contains

  integer(c_int) function neo_real_precision() bind(C, name="neo_real_precision")
    neo_real_precision = precision(1.0_qp)
  end function neo_real_precision

  subroutine body_state(body_id, et, state)
    integer(c_int), intent(in) :: body_id
    real(qp), intent(in) :: et
    real(qp), intent(out) :: state(6)
    real(c_double) :: state64(6), light_time
    character(kind=c_char), parameter :: ref(6) = [ &
         'J', '2', '0', '0', '0', c_null_char ]
    character(kind=c_char), parameter :: correction(5) = [ &
         'N', 'O', 'N', 'E', c_null_char ]

    call spkez_c(body_id, real(et, c_double), ref, correction, &
         0_c_int, state64, light_time)
    state = real(state64, qp)
  end subroutine body_state

  pure function norm3(vector) result(length)
    real(qp), intent(in) :: vector(3)
    real(qp) :: length
    length = sqrt(sum(vector * vector))
  end function norm3

  pure function cross3(a, b) result(c)
    real(qp), intent(in) :: a(3), b(3)
    real(qp) :: c(3)
    c = [a(2) * b(3) - a(3) * b(2), &
         a(3) * b(1) - a(1) * b(3), &
         a(1) * b(2) - a(2) * b(1)]
  end function cross3

  subroutine derivatives(t, y, et0, gm, options, parameters, dydt, ok)
    real(qp), intent(in) :: t, y(6), et0, gm(nbody), parameters(8)
    integer(c_int), intent(in) :: options(3)
    real(qp), intent(out) :: dydt(6)
    logical, intent(out) :: ok
    real(qp) :: acceleration(3), state(6), displacement(3), distance
    real(qp) :: sun(6), rvec(3), vvec(3), rhat(3), hhat(3), that(3)
    real(qp) :: radius, hnorm, mu, v2, rv, srp, drag(3), radial_velocity
    real(qp) :: scale, a1, a2, a3, area_mass, cr, wind_ratio
    integer :: index

    acceleration = 0.0_qp
    sun = 0.0_qp
    do index = 1, nbody
      if (gm(index) == 0.0_qp) cycle
      call body_state(body_ids(index), et0 + t, state)
      if (index == 1) sun = state
      displacement = state(1:3) - y(1:3)
      distance = norm3(displacement)
      if (distance <= 0.0_qp) then
        ok = .false.
        return
      end if
      acceleration = acceleration + gm(index) * displacement / distance**3
    end do

    rvec = y(1:3) - sun(1:3)
    vvec = y(4:6) - sun(4:6)
    radius = norm3(rvec)
    if (radius <= 0.0_qp) then
      ok = .false.
      return
    end if
    rhat = rvec / radius
    hhat = cross3(rvec, vvec)
    hnorm = norm3(hhat)
    if (hnorm <= 0.0_qp) then
      ok = .false.
      return
    end if
    hhat = hhat / hnorm
    that = cross3(hhat, rhat)
    mu = gm(1)

    if (options(1) /= 0_c_int) then
      v2 = sum(vvec * vvec)
      rv = sum(rvec * vvec)
      acceleration = acceleration + mu / (c_km_s**2 * radius**3) * &
           ((4.0_qp * mu / radius - v2) * rvec + 4.0_qp * rv * vvec)
    end if

    area_mass = parameters(1)
    cr = parameters(2)
    wind_ratio = parameters(3)
    if (area_mass > 0.0_qp) then
      srp = solar_pressure * cr * area_mass / 1000.0_qp * &
           (au_km / radius)**2
      acceleration = acceleration + srp * rhat
      if (options(2) /= 0_c_int) then
        radial_velocity = sum(vvec * rhat)
        drag = -srp * (radial_velocity * rhat + vvec) / c_km_s
        acceleration = acceleration + drag
        if (options(3) /= 0_c_int) then
          acceleration = acceleration + wind_ratio * drag
        end if
      end if
    end if

    a1 = parameters(4)
    a2 = parameters(5)
    a3 = parameters(6)
    scale = (au_km / radius)**2 * au_km / day_s**2
    acceleration = acceleration + scale * (a1 * rhat + a2 * that + a3 * hhat)

    dydt(1:3) = y(4:6)
    dydt(4:6) = acceleration
    ok = .true.
  end subroutine derivatives

  subroutine rkdp54_step(t, y, h, et0, gm, options, parameters, y5, error, ok, nfev)
    real(qp), intent(in) :: t, y(6), h, et0, gm(nbody), parameters(8)
    integer(c_int), intent(in) :: options(3)
    real(qp), intent(out) :: y5(6), error(6)
    logical, intent(out) :: ok
    integer(c_int), intent(inout) :: nfev
    real(qp) :: k1(6), k2(6), k3(6), k4(6), k5(6), k6(6), k7(6)
    real(qp) :: work(6), y4(6)

    call derivatives(t, y, et0, gm, options, parameters, k1, ok)
    if (.not. ok) return
    work = y + h * (1.0_qp / 5.0_qp) * k1
    call derivatives(t + h / 5.0_qp, work, et0, gm, options, parameters, k2, ok)
    if (.not. ok) return
    work = y + h * (3.0_qp / 40.0_qp * k1 + 9.0_qp / 40.0_qp * k2)
    call derivatives(t + h * 3.0_qp / 10.0_qp, work, et0, gm, options, parameters, k3, ok)
    if (.not. ok) return
    work = y + h * (44.0_qp / 45.0_qp * k1 - 56.0_qp / 15.0_qp * k2 + 32.0_qp / 9.0_qp * k3)
    call derivatives(t + h * 4.0_qp / 5.0_qp, work, et0, gm, options, parameters, k4, ok)
    if (.not. ok) return
    work = y + h * (19372.0_qp / 6561.0_qp * k1 - 25360.0_qp / 2187.0_qp * k2 + &
         64448.0_qp / 6561.0_qp * k3 - 212.0_qp / 729.0_qp * k4)
    call derivatives(t + h * 8.0_qp / 9.0_qp, work, et0, gm, options, parameters, k5, ok)
    if (.not. ok) return
    work = y + h * (9017.0_qp / 3168.0_qp * k1 - 355.0_qp / 33.0_qp * k2 + &
         46732.0_qp / 5247.0_qp * k3 + 49.0_qp / 176.0_qp * k4 - &
         5103.0_qp / 18656.0_qp * k5)
    call derivatives(t + h, work, et0, gm, options, parameters, k6, ok)
    if (.not. ok) return
    y5 = y + h * (35.0_qp / 384.0_qp * k1 + 500.0_qp / 1113.0_qp * k3 + &
         125.0_qp / 192.0_qp * k4 - 2187.0_qp / 6784.0_qp * k5 + &
         11.0_qp / 84.0_qp * k6)
    call derivatives(t + h, y5, et0, gm, options, parameters, k7, ok)
    if (.not. ok) return
    y4 = y + h * (5179.0_qp / 57600.0_qp * k1 + 7571.0_qp / 16695.0_qp * k3 + &
         393.0_qp / 640.0_qp * k4 - 92097.0_qp / 339200.0_qp * k5 + &
         187.0_qp / 2100.0_qp * k6 + 1.0_qp / 40.0_qp * k7)
    error = y5 - y4
    nfev = nfev + 7_c_int
  end subroutine rkdp54_step

  subroutine propagate_neo(initial64, et064, times64, nsample, gm64, &
       options, parameters64, rtol64, atol_position64, atol_velocity64, &
       max_step64, output64, nfev, status) bind(C, name="propagate_neo")
    integer(c_int), value :: nsample
    real(c_double), intent(in) :: initial64(6), et064, times64(nsample)
    real(c_double), intent(in) :: gm64(nbody), parameters64(8)
    integer(c_int), intent(in) :: options(3)
    real(c_double), intent(in) :: rtol64, atol_position64, atol_velocity64
    real(c_double), intent(in) :: max_step64
    real(c_double), intent(out) :: output64(6, nsample)
    integer(c_int), intent(out) :: nfev, status
    real(qp) :: y(6), trial(6), error(6), gm(nbody), parameters(8)
    real(qp) :: et0, target, t, h, remaining, rtol, atol(6), scale(6)
    real(qp) :: error_norm, factor, max_step
    logical :: ok
    integer :: index

    y = real(initial64, qp)
    gm = real(gm64, qp)
    parameters = real(parameters64, qp)
    et0 = real(et064, qp)
    rtol = real(rtol64, qp)
    atol(1:3) = real(atol_position64, qp)
    atol(4:6) = real(atol_velocity64, qp)
    max_step = real(max_step64, qp)
    t = real(times64(1), qp)
    h = min(max_step, max(1.0_qp, (real(times64(nsample), qp) - t) / 1000.0_qp))
    nfev = 0_c_int
    status = 0_c_int
    output64(:, 1) = real(y, c_double)

    do index = 2, nsample
      target = real(times64(index), qp)
      if (target < t) then
        status = 2_c_int
        return
      end if
      do while (t < target)
        remaining = target - t
        h = min(h, remaining, max_step)
        if (h <= 1.0e-9_qp) then
          status = 3_c_int
          return
        end if
        call rkdp54_step(t, y, h, et0, gm, options, parameters, trial, error, ok, nfev)
        if (.not. ok) then
          status = 4_c_int
          return
        end if
        scale = atol + rtol * max(abs(y), abs(trial))
        error_norm = maxval(abs(error) / scale)
        if (error_norm <= 1.0_qp) then
          t = t + h
          y = trial
          if (error_norm == 0.0_qp) then
            factor = 5.0_qp
          else
            factor = min(5.0_qp, max(0.2_qp, 0.9_qp * error_norm**(-0.2_qp)))
          end if
          h = min(max_step, h * factor)
        else
          factor = max(0.1_qp, 0.9_qp * error_norm**(-0.25_qp))
          h = h * factor
        end if
      end do
      output64(:, index) = real(y, c_double)
    end do
  end subroutine propagate_neo

end module neo_integrator
